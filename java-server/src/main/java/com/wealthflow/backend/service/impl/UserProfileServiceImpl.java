package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.repository.UserProfileRepository;
import com.wealthflow.backend.service.UserProfileService;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;
import com.wealthflow.backend.exception.ResourceNotFoundException;

import java.util.List;

@Service
@Transactional
public class UserProfileServiceImpl implements UserProfileService {

    private final UserProfileRepository userProfileRepository;

    public UserProfileServiceImpl(UserProfileRepository userProfileRepository) {
        this.userProfileRepository = userProfileRepository;
    }

    @Override
    public UserProfile createUserProfile(UserProfile userProfile) {
        return userProfileRepository.save(userProfile);
    }

    @Override
    public UserProfile getUserProfileById(Long id) {
        return userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found with id: " + id));
    }

    @Override
    public List<UserProfile> getAllUserProfiles() {
        return userProfileRepository.findAll();
    }

    @Override
    public void deleteUserProfile(Long id) {
        if (!userProfileRepository.existsById(id)) {
            throw new ResourceNotFoundException("UserProfile not found with id: " + id);
        }
        userProfileRepository.deleteById(id);
    }
}
