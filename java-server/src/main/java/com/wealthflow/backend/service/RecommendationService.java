package com.wealthflow.backend.service;

import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.model.UserProfile;

public interface RecommendationService {
    Recommendation generateRecommendation(UserProfile userProfile);
}
