from app import create_app


def main():
    # Initialize the Flask application
    application_instance = create_app()
    # Start the Flask application server
    application_instance.run()


if __name__ == "__main__":
    main()
