from loguru import logger
from langaug import setup_logging


def main() -> None:
    setup_logging("INFO")
    logger.info("langaug initialized")


if __name__ == "__main__":
    main()
