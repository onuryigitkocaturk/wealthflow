package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;

public interface UserProfileService {

    UserProfileResponse createProfile(UserProfileRequest request);

    UserProfileResponse getProfile(Long id);

    UserProfileResponse updateProfile(Long id, UserProfileRequest request);
}
