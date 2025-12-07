package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.RecommendationRequest;
import com.wealthflow.backend.dto.RecommendationResponse;
import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.model.UserProfile;

public interface RecommendationService {
    RecommendationResponse generateRecommendation(RecommendationRequest request);
    RecommendationResponse getById(Long id);
}
