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

"""Run ament_flake8 when the ROS lint tools are installed."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from ament_flake8.main import main_with_errors
except ImportError:
    main_with_errors = None


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    if main_with_errors is None:
        pytest.skip('ament_flake8 is not installed')
    root = str(Path(__file__).resolve().parents[1])
    rc, errors = main_with_errors(argv=[root])
    message = 'Found %d code style errors / warnings:\n' % len(errors)
    assert rc == 0, message + '\n'.join(errors)
