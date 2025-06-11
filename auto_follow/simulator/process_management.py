import subprocess

import psutil


class ProcessManager:
    """Handles process-related operations"""

    @staticmethod
    def verify_process_running(process: subprocess.Popen[bytes] | None, name: str) -> bool:
        """
        Verify if a process is still running and check its output.

        @param process: Process to check.
        @param name: Process name for logging.
        @return: True if the process is still running, False otherwise.
        """
        if process is None:
            return False

        if process.poll() is not None:
            error_output = process.stderr.read() if process.stderr else "No error output available"
            print(f"Process '{name}' terminated unexpectedly. Error: {error_output}")
            return False

        return True

    @staticmethod
    def clean_remaining_processes(name: str) -> None:
        """Kill any remaining processes by its name."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if name.lower() in proc.info['name'].lower():
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"Cannot kill process '{name}': Access Denied or process not found.")

    @staticmethod
    def kill_process_tree(pid: int | None) -> None:
        """Kill a process (given its ID) and all its children recursively."""
        if pid is None:
            return

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass


class SphinxCommandManager:
    """Handles sphinx and drone firmware related operations"""

    @staticmethod
    def drop_all_instances():
        try:
            subprocess.run(
                ['fdc', 'drop_all', 'instances'],
                check=True,
                capture_output=True,
                text=True
            )
            print("Successfully dropped all FirmwareD Client instances")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to drop FirmwareD Client instances: stderr:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error: {e!s}")
            return False

    @staticmethod
    def restart_firmwared(sudo_password):
        try:
            process = subprocess.Popen(
                ["sudo", '-S', "systemctl", "restart", "firmwared.service"],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            _ = process.communicate(sudo_password + '\n')[1]

            if process.returncode == 0:
                print("Service restarted successfully")
                return True
            else:
                print("Unable to restart service...")
                return False
        except subprocess.CalledProcessError as e:
            print(f"Failed to restart service: stderr:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error encountered when restarting firmwared service: {e!s}")
            return False
