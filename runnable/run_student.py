from auto_follow.controllers.distilled_network_controller import main_distilled_student_controller


def main():
    main_distilled_student_controller()


if __name__ == "__main__":
    """
    Experiment names:
        real-student-down-left
        real-student-down-right
        real-student-front-small-offset-right
        real-student-front-small-offset-left
        real-student-left
        real-student-right
        real-student-up-left
        real-student-up-right
    """
    main()