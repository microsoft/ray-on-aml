import logging
import os

instrumentation_key = os.environ.get("APPINSIGHTS_INSTRUMENTATION_KEY", "")

class _EventLogger:

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(__name__).getChild(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    
        try:
            from opencensus.ext.azure.log_exporter import AzureEventHandler

            # Doc: Set up Azure Monitor for your Python application
            # https://learn.microsoft.com/en-us/azure/azure-monitor/app/opencensus-python#send-events
            if instrumentation_key and not _EventLogger._found_handler(logger, AzureEventHandler):
                logger.addHandler(
                    AzureEventHandler(
                        connection_string="InstrumentationKey=" + instrumentation_key
                    )
                )
        except ImportError:
            pass

        return logger
    
    @staticmethod
    def track_event(logger, name, properties=None):
        custom_dimensions = _EventLogger._try_get_run_info()
        if properties is not None:
            custom_dimensions.update(properties)
            
        logger.info(name, extra={"custom_dimensions": custom_dimensions})

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
            import ray

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
            "ray_version": ray.__version__,
        }
