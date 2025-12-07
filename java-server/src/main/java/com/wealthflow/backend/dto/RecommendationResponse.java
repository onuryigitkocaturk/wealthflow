package com.wealthflow.backend.dto;

import java.time.LocalDateTime;
import java.util.List;

public record RecommendationResponse(
        Long id,
        Long userProfileId,
        String recommendationSummary,
        Double expectedReturn,
        List<AllocationResponse> allocations,
        LocalDateTime createdAt
) {}
