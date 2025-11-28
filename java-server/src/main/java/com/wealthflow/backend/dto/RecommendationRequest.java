package com.wealthflow.backend.dto;

import jakarta.validation.constraints.NotNull;

public class RecommendationRequest {

    @NotNull(message = "User profile id is required")
    private Long userProfileId;

    public Long getUserProfileId() {
        return userProfileId;
    }
    public void setUserProfileId(Long userProfileId) {
        this.userProfileId = userProfileId;
    }
}
