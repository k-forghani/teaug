import logging

from langaug import setup_logging


def main() -> None:
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("langaug initialized")


if __name__ == "__main__":
    main()
