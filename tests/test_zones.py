from wrongway.zones import OK, PENDING, WRONG, ZoneValidator

FRAME = (100, 100)

AREAS = {
    "A": [(0.0, 0.0), (0.3, 0.0), (0.3, 1.0), (0.0, 1.0)],   # left third
    "B": [(0.35, 0.0), (0.65, 0.0), (0.65, 1.0), (0.35, 1.0)],  # middle
    "C": [(0.7, 0.0), (1.0, 0.0), (1.0, 1.0), (0.7, 1.0)],   # right third
}


def make_validator():
    return ZoneValidator(AREAS, wrong_entries=["C"], wrong_transitions=[("B", "C")])


def test_correct_path_a_to_c_is_ok():
    validator = make_validator()
    assert validator.update(1, (10, 50), FRAME) == PENDING   # enters via A
    assert validator.update(1, (50, 50), FRAME) == OK        # crosses B
    assert validator.update(1, (90, 50), FRAME) == OK        # exits via C
    assert 1 not in validator.flagged


def test_wrong_entry_is_immediately_wrong():
    validator = make_validator()
    assert validator.update(2, (90, 50), FRAME) == WRONG     # appears in C
    assert 2 in validator.flagged


def test_wrong_transition_b_to_c():
    validator = make_validator()
    assert validator.update(3, (50, 50), FRAME) == PENDING   # enters via B
    assert validator.update(3, (90, 50), FRAME) == WRONG     # reaches C
    assert 3 in validator.flagged


def test_outside_all_zones_is_pending():
    validator = make_validator()
    assert validator.update(4, (32, 50), FRAME) == PENDING   # in the gap
    assert validator.entry_zone.get(4) is None


def test_flag_is_sticky():
    validator = make_validator()
    validator.update(5, (90, 50), FRAME)
    assert validator.update(5, (10, 50), FRAME) == WRONG


def test_forget_clears_state():
    validator = make_validator()
    validator.update(6, (10, 50), FRAME)
    validator.forget(6)
    assert 6 not in validator.entry_zone
