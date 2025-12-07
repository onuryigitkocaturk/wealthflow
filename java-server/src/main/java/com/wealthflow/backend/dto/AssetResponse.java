package com.wealthflow.backend.dto;

import com.wealthflow.backend.model.enums.AssetType;
import com.wealthflow.backend.model.enums.RiskLevel;

public record AssetResponse(
        Long id,
        String symbol,
        String name,
        AssetType type,
        RiskLevel riskLevel,
        Double esgScore
) {}
