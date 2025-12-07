package com.wealthflow.backend.mapper;

import com.wealthflow.backend.dto.AllocationResponse;
import com.wealthflow.backend.dto.RecommendationResponse;
import com.wealthflow.backend.model.Recommendation;

import java.util.List;

public class RecommendationMapper {

    public static RecommendationResponse toResponse(Recommendation entity) {

        List<AllocationResponse> allocationResponses = entity.getAllocations()
                .stream()
                .map(AllocationMapper::toResponse)
                .toList();

        return new RecommendationResponse(
                entity.getId(),
                entity.getUserProfile().getId(),
                null,
                null,
                allocationResponses,
                entity.getGeneratedAt()
        );
    }
}
