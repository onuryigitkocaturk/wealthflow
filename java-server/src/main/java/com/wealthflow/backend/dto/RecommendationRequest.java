package com.wealthflow.backend.dto;

import com.wealthflow.backend.model.enums.RiskLevel;
import jakarta.validation.constraints.NotNull;


public record RecommendationRequest(
        @NotNull(message = "User profile ID is required")
        Long userProfileId,

        @NotNull(message = "Manueal override flag must be provided")
        Boolean manualOverride,

        RiskLevel overrideRiskLevel
) {}
