# Copyright 2026 Arpan Bom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run ament_pep257 when the ROS lint tools are installed."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from ament_pep257.main import main
except ImportError:
    main = None


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    if main is None:
        pytest.skip('ament_pep257 is not installed')
    root = str(Path(__file__).resolve().parents[1])
    assert main(argv=[root]) == 0, 'Found docstring style errors'
