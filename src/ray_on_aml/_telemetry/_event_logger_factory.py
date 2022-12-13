import logging

instrumentation_key = "28f3e437-7871-4f33-a75a-b5b3895438db"

class _LoggerFactory:

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(__name__).getChild(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    
        try:
            from opencensus.ext.azure.log_exporter import AzureEventHandler

            # Doc: Set up Azure Monitor for your Python application
            # https://learn.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python#send-events
            if not _LoggerFactory._found_handler(logger, AzureEventHandler):
                logger.addHandler(
                    AzureEventHandler(
                        connection_string="InstrumentationKey=" + instrumentation_key
                    )
                )
        except ImportError:
            pass

        return logger

    @staticmethod
    def _found_handler(logger, handler_type):
        for log_handler in logger.handlers:
            if isinstance(log_handler, handler_type):
                return True
        return False

    @staticmethod
    def _try_get_run_info():
        try:
            import re
            import os

            location = os.environ.get("AZUREML_SERVICE_ENDPOINT")
            location = re.compile("//(.*?)\\.").search(location).group(1)
        except Exception:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
        return {
            "subscription": os.environ.get("AZUREML_ARM_SUBSCRIPTION", ""),
            "run_id": os.environ.get("AZUREML_RUN_ID", ""),
            "resource_group": os.environ.get("AZUREML_ARM_RESOURCEGROUP", ""),
            "workspace_name": os.environ.get("AZUREML_ARM_WORKSPACE_NAME", ""),
            "experiment_id": os.environ.get("AZUREML_EXPERIMENT_ID", ""),
            "location": location,
        }
