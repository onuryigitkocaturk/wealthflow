package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.RecommendationRequest;
import com.wealthflow.backend.dto.RecommendationResponse;

public interface RecommendationService {
    RecommendationResponse generateRecommendation(RecommendationRequest request);
    RecommendationResponse getById(Long id);
}
