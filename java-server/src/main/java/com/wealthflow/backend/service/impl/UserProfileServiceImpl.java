package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;
import com.wealthflow.backend.exception.ResourceNotFoundException;
import com.wealthflow.backend.mapper.UserProfileMapper;
import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.repository.UserProfileRepository;
import com.wealthflow.backend.service.UserProfileService;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Transactional
public class UserProfileServiceImpl implements UserProfileService {

    private final UserProfileRepository userProfileRepository;

    public UserProfileServiceImpl(UserProfileRepository userProfileRepository) {
        this.userProfileRepository = userProfileRepository;
    }

    @Override
    public UserProfileResponse createProfile(UserProfileRequest request) {
        UserProfile profile = new UserProfile();
        profile.setEmail(request.email());
        profile.setName(request.name());
        profile.setAge(request.age());
        profile.setAnnualIncome(request.annualIncome());
        profile.setInvestmentHorizon(request.investmentHorizon());
        profile.setEsgPreference(request.esgPreference());
        profile.setRiskTolerance(request.riskTolerance());

        UserProfile saved = userProfileRepository.save(profile);

        return UserProfileMapper.toResponse(saved);
    }

    @Override
    public UserProfileResponse getProfile(Long id) {
        UserProfile profile = userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found: " + id));

        return UserProfileMapper.toResponse(profile);
    }

    @Override
    public UserProfileResponse updateProfile(Long id, UserProfileRequest request) {
        UserProfile profile = userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found: " + id));

        profile.setEmail(request.email());
        profile.setName(request.name());
        profile.setAge(request.age());
        profile.setAnnualIncome(request.annualIncome());
        profile.setInvestmentHorizon(request.investmentHorizon());
        profile.setEsgPreference(request.esgPreference());
        profile.setRiskTolerance(request.riskTolerance());

        UserProfile updated = userProfileRepository.save(profile);

        return UserProfileMapper.toResponse(updated);
    }

    @Override
    public List<UserProfileResponse> getAllProfiles() {
        return userProfileRepository.findAll()
                .stream()
                .map(UserProfileMapper::toResponse)
                .toList();
    }

    @Override
    public void deleteProfile(Long id) {
        UserProfile profile = userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found: " + id));

        userProfileRepository.delete(profile);
    }

}
