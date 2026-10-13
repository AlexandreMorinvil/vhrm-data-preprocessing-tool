import unittest

from PyQt6.QtCore import QCoreApplication, QTimer

from app.ffmpeg_utils import FFmpegWorker


class WorkerLifetimeTests(unittest.TestCase):
    def test_handler_may_drop_the_worker_while_run_is_returning(self) -> None:
        """Regression: the app aborted with "QThread: Destroyed while thread is still running"."""
        app = QCoreApplication.instance() or QCoreApplication([])
        state = {"worker": None, "runs": 0}

        def job(worker):
            sum(i * i for i in range(5000))

        def launch():
            worker = FFmpegWorker(job)
            worker.finished.connect(finished)
            state["worker"] = worker  # the only reference, as in the mode widgets
            worker.start()

        def finished(ok, _message):
            self.assertTrue(ok)
            state["worker"] = None
            state["runs"] += 1
            if state["runs"] < 300:
                launch()
            else:
                app.quit()

        QTimer.singleShot(0, launch)
        QTimer.singleShot(60_000, app.quit)
        app.exec()
        self.assertEqual(state["runs"], 300)


if __name__ == "__main__":
    unittest.main()
