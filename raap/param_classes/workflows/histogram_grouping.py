

from pydantic import BaseModel

from param_classes.workflows.input_output import GroupInfoParams, OutputParams


class HistogramGroupingWorkflowConfig(BaseModel):
    """
    Workflow for grouping and aggregating histogram results.
    """
    grouping: GroupInfoParams = GroupInfoParams()
    palette: dict = {}
    output: OutputParams = OutputParams()
