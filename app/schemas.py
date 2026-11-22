"""Skema request/response API.

Nilai kategori dibatasi lewat Enum supaya input di luar domain dataset
ditolak dengan 422, bukan diam-diam dianggap vektor nol.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Gender(str, Enum):
    female = "Female"
    male = "Male"


class MultipleLines(str, Enum):
    no = "No"
    yes = "Yes"
    no_phone_service = "No phone service"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class InternetAddon(str, Enum):
    no = "No"
    yes = "Yes"
    no_internet_service = "No internet service"


class Contract(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """Satu pelanggan. Nama field sengaja sama dengan kolom dataset."""

    model_config = ConfigDict(extra="forbid")

    customerID: str | None = Field(default=None, max_length=64, description="Opsional, untuk pelacakan")
    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(gt=0, le=200)
    # Batas atas mengikuti domain: tenure maksimum 72 bulan x MonthlyCharges maksimum 200.
    TotalCharges: float = Field(ge=0, le=15000)


class PredictionResponse(BaseModel):
    customer_id: str | None
    churn_probability: float
    churn_prediction: int
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    trained_at: str | None
    threshold: float
    selection_reason: str | None
    training_rows: int | None
    environment: dict
    test_metrics: dict | None
