package com.wealthflow.backend.dto;

import com.wealthflow.backend.model.enums.RiskLevel;

public record RecommendationRequest(
        Long userProfileId,
        Boolean manualOverride,
        RiskLevel overrideRiskLevel
) {}
