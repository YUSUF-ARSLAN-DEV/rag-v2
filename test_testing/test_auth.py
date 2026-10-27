import time

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import auth


def creds(token):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_correct_password_verifies_and_wrong_one_does_not():
    hashed = auth.hash_password("demopass123")
    assert hashed != "demopass123"
    assert auth.verify_password("demopass123", hashed)
    assert not auth.verify_password("wrong-password", hashed)


def test_same_password_gives_different_hashes():
    # bcrypt adds a random salt, so two users with the same password get different hashes
    assert auth.hash_password("demopass123") != auth.hash_password("demopass123")


def test_token_round_trip_returns_the_user_id():
    token = auth.create_token(42)
    assert auth.get_current_user(creds(token)) == 42


def test_expired_token_is_rejected():
    expired = jwt.encode({"sub": "42", "exp": int(time.time()) - 10}, auth.JWT_SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as err:
        auth.get_current_user(creds(expired))
    assert err.value.status_code == 401


def test_token_signed_with_another_secret_is_rejected():
    forged = jwt.encode({"sub": "42", "exp": int(time.time()) + 60}, "some-other-secret-that-is-long-enough", algorithm="HS256")
    with pytest.raises(HTTPException) as err:
        auth.get_current_user(creds(forged))
    assert err.value.status_code == 401
