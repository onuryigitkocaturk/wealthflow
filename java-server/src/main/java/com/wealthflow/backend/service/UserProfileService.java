package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;

import java.util.List;

public interface UserProfileService {

    UserProfileResponse createProfile(UserProfileRequest request);

    UserProfileResponse getProfileByEmail(String email);

    UserProfileResponse updateProfileByEmail(String email, UserProfileRequest request);

    List<UserProfileResponse> getAllProfiles();

    void deleteProfile(Long id);

}
