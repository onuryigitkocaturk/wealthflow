package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;
import com.wealthflow.backend.exception.ResourceNotFoundException;
import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.model.enums.RiskTolerance;
import com.wealthflow.backend.repository.UserProfileRepository;
import com.wealthflow.backend.service.UserProfileService;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

@Service
@Transactional
public class UserProfileServiceImpl implements UserProfileService {

    private final UserProfileRepository userProfileRepository;

    public UserProfileServiceImpl(UserProfileRepository userProfileRepository) {
        this.userProfileRepository = userProfileRepository;
    }

    // CREATE
    @Override
    public UserProfileResponse createProfile(UserProfileRequest request) {

        UserProfile profile = new UserProfile();
        profile.setEmail(request.getEmail());
        profile.setName(request.getName());
        profile.setAge(request.getAge());
        profile.setAnnualIncome(request.getAnnualIncome());
        profile.setInvestmentHorizon(request.getInvestmentHorizon());
        profile.setEsgPreference(request.getEsgPreference());
        profile.setRiskTolerance(RiskTolerance.valueOf(request.getRiskTolerance().toUpperCase()));

        UserProfile saved = userProfileRepository.save(profile);

        return toResponse(saved);
    }

    // GET
    @Override
    public UserProfileResponse getProfile(Long id) {
        UserProfile profile = userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found: " + id));

        return toResponse(profile);
    }

    // UPDATE
    @Override
    public UserProfileResponse updateProfile(Long id, UserProfileRequest request) {

        UserProfile profile = userProfileRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("UserProfile not found: " + id));

        profile.setEmail(request.getEmail());
        profile.setName(request.getName());
        profile.setAge(request.getAge());
        profile.setAnnualIncome(request.getAnnualIncome());
        profile.setInvestmentHorizon(request.getInvestmentHorizon());
        profile.setEsgPreference(request.getEsgPreference());
        profile.setRiskTolerance(RiskTolerance.valueOf(request.getRiskTolerance().toUpperCase()));

        UserProfile updated = userProfileRepository.save(profile);

        return toResponse(updated);
    }

    // Mapping helper
    private UserProfileResponse toResponse(UserProfile profile) {
        UserProfileResponse dto = new UserProfileResponse();

        dto.setId(profile.getId());
        dto.setEmail(profile.getEmail());
        dto.setName(profile.getName());
        dto.setAge(profile.getAge());
        dto.setAnnualIncome(profile.getAnnualIncome());
        dto.setInvestmentHorizon(profile.getInvestmentHorizon());
        dto.setEsgPreference(profile.getEsgPreference());
        dto.setRiskTolerance(profile.getRiskTolerance().toString());
        dto.setRiskOverride(profile.getRiskOverride());
        dto.setOverrideTolerance(profile.getOverrideTolerance() == null ? null : profile.getOverrideTolerance().toString());
        dto.setRiskScore(profile.getRiskScore());

        return dto;
    }
}
