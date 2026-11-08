import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from tests import synthetic as S


@pytest.fixture(scope="session")
def scene():
    return S.scene((256, 256), seed=7)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.analyze import app

    return TestClient(app)


def gray(a):
    from backend.imgops import to_gray

    return to_gray(a)
