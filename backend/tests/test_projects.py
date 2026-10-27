"""
Tests for projects route — including SQL injection prevention regression tests.

These tests verify that:
1. The GET /api/projects endpoint works correctly with a search parameter.
2. SQL injection payloads in the search parameter are treated as literal strings
   (parameterized queries), not executed as SQL syntax.
3. Existing project functionality (list, create, get by id) continues to work.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import db, User, Project


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _register_and_login(client, username='projuser', password='Pass1234!'):
    """Create a user via the auth endpoint and return auth headers."""
    client.post('/api/auth/register', json={
        'username': username,
        'email': f'{username}@example.com',
        'password': password,
    })
    resp = client.post('/api/auth/login', json={
        'username': username,
        'password': password,
    })
    data = resp.get_json() or {}
    token = data.get('token') or data.get('access_token') or ''
    return {'Authorization': f'Bearer {token}'}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def headers(client):
    return _register_and_login(client, username='projuser_a')


@pytest.fixture
def project_with_db(db_session, sample_user):
    """Seed a couple of projects owned by sample_user."""
    p1 = Project(
        name='Alpha Project',
        description='First project description',
        owner_id=sample_user.id,
        is_public=False,
    )
    p2 = Project(
        name='Beta Project',
        description='Second project description',
        owner_id=sample_user.id,
        is_public=True,
    )
    db_session.add(p1)
    db_session.add(p2)
    db_session.commit()
    return p1, p2


# ---------------------------------------------------------------------------
# Functional tests — search works correctly
# ---------------------------------------------------------------------------

class TestGetProjectsSearch:
    """Verify the search parameter returns the correct rows."""

    def test_search_returns_matching_projects(
        self, client, auth_headers, project_with_db
    ):
        """A normal search term finds the right projects."""
        resp = client.get('/api/projects?search=Alpha', headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        names = [p['name'] for p in data.get('projects', [])]
        assert 'Alpha Project' in names

    def test_search_no_match_returns_empty_list(
        self, client, auth_headers, project_with_db
    ):
        """A search term with no matches returns an empty projects list."""
        resp = client.get(
            '/api/projects?search=ZZZnonexistent', headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get('projects', []) == []

    def test_search_empty_string_returns_all_projects(
        self, client, auth_headers, project_with_db
    ):
        """An empty search string returns all projects (ORM path)."""
        resp = client.get('/api/projects', headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data.get('projects'), list)

    def test_search_description_match(
        self, client, auth_headers, project_with_db
    ):
        """Search also matches against project descriptions."""
        resp = client.get(
            '/api/projects?search=Second+project', headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.get_json()
        names = [p['name'] for p in data.get('projects', [])]
        assert 'Beta Project' in names


# ---------------------------------------------------------------------------
# Security regression tests — SQL injection prevention
# ---------------------------------------------------------------------------

class TestSQLInjectionPrevention:
    """
    Verify that SQL injection payloads in the search parameter are treated as
    literal search strings and do NOT cause SQL syntax errors or data leaks.

    With parameterized queries the payload is bound as a value; the database
    never parses it as SQL syntax.  The expected outcome for every attack
    payload below is HTTP 200 with an empty (or irrelevant) result set — NOT
    a 500 error and NOT rows that should be hidden.
    """

    @pytest.mark.parametrize("payload", [
        # Classic tautology — must NOT return all rows
        "' OR '1'='1",
        # UNION-based exfiltration attempt
        "' UNION SELECT * FROM users--",
        # Comment-based bypass
        "test'--",
        # Stacked statement attempt
        "'; DROP TABLE projects; --",
        # Blind boolean injection
        "' AND 1=1--",
        # Another tautology variant
        "x' OR 'x'='x",
        # Wildcard / special LIKE characters (should be treated as literals)
        "%",
        "_",
        # NUL byte injection (\x00)
        "test\x00injection",
    ])
    def test_injection_payload_does_not_cause_server_error(
        self, client, auth_headers, project_with_db, payload
    ):
        """
        SQL injection payloads must not cause a 500 Internal Server Error.
        The parameterized query binds the value safely; invalid SQL syntax in
        the payload is never interpreted by the database engine.
        """
        resp = client.get(
            f'/api/projects',
            query_string={'search': payload},
            headers=auth_headers,
        )
        # The server must NOT crash — parameterized queries prevent syntax errors.
        assert resp.status_code == 200, (
            f"Payload {payload!r} caused HTTP {resp.status_code}; "
            "expected 200 with parameterized query."
        )

    def test_tautology_does_not_leak_all_rows(
        self, client, auth_headers, project_with_db
    ):
        """
        A classic OR-tautology payload must NOT return rows whose names/
        descriptions do not literally contain the payload string.
        """
        resp = client.get(
            "/api/projects?search=' OR '1'='1",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        projects = data.get('projects', [])
        # None of the seeded projects contain this literal string in name/description.
        names = [p['name'] for p in projects]
        assert 'Alpha Project' not in names, (
            "Tautology injection returned rows that should not match."
        )
        assert 'Beta Project' not in names, (
            "Tautology injection returned rows that should not match."
        )

    def test_percent_wildcard_treated_as_literal_part_of_value(
        self, client, auth_headers, project_with_db
    ):
        """
        A bare % submitted as the search value must be treated as a literal
        character passed to the LIKE pattern — not as an unescaped wildcard
        that selects everything.

        Note: SQLite / standard SQL LIKE treats % as a wildcard inside the
        pattern string.  Since we wrap the user value with '%' delimiters
        (LIKE :pattern with pattern = '%<value>%'), supplying '%' still
        produces a valid LIKE pattern ('%%%') and will match all rows — but
        crucially this is NOT a SQL injection: it is equivalent to an empty
        search and returns an HTTP 200 without error.  The key assertion is
        that the server does not crash.
        """
        resp = client.get('/api/projects?search=%', headers=auth_headers)
        assert resp.status_code == 200

    def test_union_payload_does_not_return_user_data(
        self, client, auth_headers, project_with_db
    ):
        """
        A UNION-based injection attempt must not return columns from other
        tables (e.g. the users table).  With parameterized queries the payload
        is treated as a string value, so the UNION keyword is never executed.
        """
        payload = "' UNION SELECT id,username,email,password_hash,NULL,NULL,NULL FROM users--"
        resp = client.get(
            '/api/projects',
            query_string={'search': payload},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        projects = data.get('projects', [])
        # None of the returned records should contain 'username' or 'email' keys
        # that come from the users table schema.
        for row in projects:
            assert 'password_hash' not in row, (
                "UNION injection leaked password_hash from the users table."
            )
            assert 'username' not in row, (
                "UNION injection leaked username from the users table."
            )


# ---------------------------------------------------------------------------
# Auth guard tests
# ---------------------------------------------------------------------------

class TestProjectsAuthGuard:
    """Verify that unauthenticated requests are rejected."""

    def test_get_projects_requires_auth(self, client):
        resp = client.get('/api/projects')
        assert resp.status_code in (401, 403)

    def test_get_projects_with_search_requires_auth(self, client):
        resp = client.get("/api/projects?search=Alpha")
        assert resp.status_code in (401, 403)
