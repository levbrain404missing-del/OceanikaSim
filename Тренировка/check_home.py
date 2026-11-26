"""Полная домашняя проверка. Обновляет отчёт и возвращает код 1 при неполном прогоне."""
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import NamedTemporaryFile
import unittest


HERE = Path(__file__).resolve().parent
REPORT = HERE / "Результат домашней проверки.md"


def required_cases():
    geo = "test_geometry_missions.GeometricMissionTests.test_"
    cam = "test_visual_mujoco.MovingCameraTests.test_"
    model = "test_physics.PhysicsBench.test_"
    replay = "test_replay.ReplayTests.test_"
    orbits = (geo+"circle_and_eight_do_full_orbits_and_land", geo+"measured_path_stays_in_tube_and_turns_in_opposite_directions")
    landing = (cam+"landing_footprint_in_random_zone_or_requested_marker",cam+"landing_finishes_on_ground_with_low_speed")
    loss = cam+"camera_dropout_at_second_read"
    return [
        ("1. Круг в XZ",orbits),
        ("2. Восьмёрка в XZ",orbits),
        ("3. Яркая посадочная зона",(*landing,loss)),
        ("4. Замкнутая линия",(cam+"full_lap_and_motion_beyond_start_until_external_stop",
                            cam+"lost_line_during_a_move_stops_and_records_failed_run",loss)),
        ("5. Повороты и финиш по ArUco",(cam+"two_marker_turns_and_final_touchdown",loss)),
        ("6. Цветные ворота",(cam+"crosses_requested_3d_gate_with_clearance",
                             cam+"judge_rejects_wrong_gate_despite_wrong_positive_detections",loss)),
        ("7. Посадка по выбранному ID",(*landing,cam+"unavailable_mark_and_wrong_color",loss)),
        ("8. Кратчайший маршрут",(model+"route_returns_after_three_points_and_lands",model+"event_loss_stops_route_and_lands")),
        ("9. Сферические координаты",(geo+"spherical_target_is_relative_to_launch_not_airborne_height",
                                      geo+"hold_needs_three_consecutive_seconds_after_telemetry_dropout")),
        ("10. Телеметрия",(replay+"telemetry_terminal_two_requests_then_disconnect",
                           "test_diagnostics.SDKTelemetryBench.test_real_sdk_getters_convert_raw_fields_before_display")),
        ("11. Карта",(replay+"map_window_receives_two_distinct_positions_without_hardware",
                     model+"rendered_trace_follows_clocked_physics_sample_under_two_seconds")),
    ]


class TestResults(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.passed = set()

    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.add(test.id())


def check(dest,suite,dependencies_ok):
    result = unittest.TextTestRunner(verbosity=1,buffer=True,resultclass=TestResults).run(suite)
    ok = bool(dependencies_ok and result.wasSuccessful() and len(result.passed) == result.testsRun and result.testsRun)
    rows, missing = [], []
    for name, expected in required_cases():
        absent = [name for name in expected if name not in result.passed]
        missing.extend(absent)
        rows.append(f"| {name} | {'Требует проверки' if absent else 'Пройдено'} |")
    ok = ok and not missing
    packages = []
    for name in ("pioneer_sdk2","mujoco","opencv-python","numpy"):
        try:
            packages.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            packages.append(f"{name} отсутствует")
    lines = ["# Результат домашней проверки","",
             f"**{'Пройдено' if ok else 'Требует проверки'}**. Успешных тестов {len(result.passed)} из {result.testsRun}.",
             "",f"Время окончания прогона {datetime.now().astimezone().isoformat(timespec='seconds')}.",
             "",f"Python {platform.python_version()}. "+", ".join(packages)+".",
             "",f"Проверка зависимостей pip {'прошла' if dependencies_ok else 'завершилась ошибкой'}.","",
             "| Задание | Проверки домашнего сценария |","|---|---|",*rows,"",
             "№ 1, 2, 8, 9 имеют режим SDK. Для № 3–7 эта таблица отражает прохождение в MuJoCo с подвижной камерой. "
             "Для № 10–11 проверены обработка показаний SDK и карта. Связь с выданным бортом, полёт и физический экран требуют испытания на месте."]
    if missing:
        lines += ["","## Обязательные проверки без успешного результата","",*sorted(set(missing))]
    details = [(str(test),trace) for test,trace in result.failures+result.errors]
    details.extend((str(test),reason) for test,reason in result.skipped)
    for name,detail in details:
        lines += ["",f"### {name}","","```text", detail.rstrip(),"```"]
    with NamedTemporaryFile("w",dir=dest.parent,delete=False,encoding="utf-8") as output:
        output.write("\n".join(lines)+"\n")
    Path(output.name).replace(dest)
    return ok


if __name__ == "__main__":
    deps = subprocess.run([sys.executable,"-B","-m","pip","check"])
    suite = unittest.defaultTestLoader.discover(str(HERE),pattern="test_*.py")
    passed = check(REPORT,suite,deps.returncode == 0)
    print(f"{'Пройдено' if passed else 'Требует проверки'}. {REPORT}")
    raise SystemExit(0 if passed else 1)
