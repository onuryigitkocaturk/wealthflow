package com.wealthflow.backend.dto;

import com.wealthflow.backend.model.enums.RiskTolerance;
import jakarta.validation.constraints.*;

public record UserProfileRequest(

        @Email(message = "Email is invalid")
        @NotBlank(message = "Email is required")
        String email,

        @NotBlank(message = "Name is required")
        String name,

        @NotNull(message = "Age is required")
        @Min(value = 18, message = "Age must be at least 18")
        Integer age,

        @NotNull(message = "Annual income is required")
        @Positive(message = "Annual income must be positive")
        Double annualIncome,

        @NotNull(message = "Investment horizon is required")
        @Positive(message = "Investment horizon must be positive")
        Integer investmentHorizon,

        @NotNull(message = "ESG preference must be provided")
        Boolean esgPreference,

        @NotNull(message = "Risk tolerance is required")
        RiskTolerance riskTolerance

) {}
