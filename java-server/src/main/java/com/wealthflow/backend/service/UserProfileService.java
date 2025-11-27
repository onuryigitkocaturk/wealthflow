package com.wealthflow.backend.service;

import com.wealthflow.backend.model.UserProfile;
import java.util.List;

public interface UserProfileService {

    UserProfile createUserProfile(UserProfile userProfile);
    UserProfile getUserProfileById(Long id);
    List<UserProfile> getAllUserProfiles();
    void deleteUserProfile(Long id);

}
