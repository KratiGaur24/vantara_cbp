from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
  """Pydantic model validating incoming raw customer JSON request payloads."""

  recency: float = Field(
      ..., ge=0.0, description="Days since last purchase (must be >= 0)"
  )
  frequency: int = Field(
      ..., ge=1, description="Total number of unique purchases (must be >= 1)"
  )
  total_spend: float = Field(
      ..., ge=0.0, description="Total monetary spend in dollars (must be >= 0)"
  )
  avg_basket_size: float = Field(
      ..., ge=0.0, description="Average items per order (must be >= 0)"
  )
  engagement_score: float = Field(
      ..., description="Engineered composite loyalty/engagement score"
  )

  model_config = {
      "json_schema_extra": {
          "example": {
              "recency": 45.0,
              "frequency": 12,
              "total_spend": 1250.50,
              "avg_basket_size": 3.2,
              "engagement_score": 18.50,
          }
      }
  }


class PredictionResponse(BaseModel):
  """Pydantic model defining the structured prediction API response."""

  churn_probability: float
  churn_prediction: int
  risk_status: str