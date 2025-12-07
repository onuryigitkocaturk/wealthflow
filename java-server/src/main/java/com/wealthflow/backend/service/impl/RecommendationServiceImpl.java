package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.dto.RecommendationRequest;
import com.wealthflow.backend.dto.RecommendationResponse;
import com.wealthflow.backend.exception.ResourceNotFoundException;
import com.wealthflow.backend.mapper.RecommendationMapper;
import com.wealthflow.backend.model.Allocation;
import com.wealthflow.backend.model.Asset;
import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.repository.AllocationRepository;
import com.wealthflow.backend.repository.AssetRepository;
import com.wealthflow.backend.repository.RecommendationRepository;
import com.wealthflow.backend.repository.UserProfileRepository;
import com.wealthflow.backend.service.RecommendationService;
import jakarta.annotation.Resource;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class RecommendationServiceImpl implements RecommendationService {

    private final AssetRepository assetRepository;
    private final RecommendationRepository recommendationRepository;
    private final AllocationRepository allocationRepository;
    private final UserProfileRepository userProfileRepository;

    public RecommendationServiceImpl(
            AssetRepository assetRepository,
            RecommendationRepository recommendationRepository,
            AllocationRepository allocationRepository,
            UserProfileRepository userProfileRepository) {
        this.assetRepository = assetRepository;
        this.recommendationRepository = recommendationRepository;
        this.allocationRepository = allocationRepository;
        this.userProfileRepository = userProfileRepository;
    }

    @Override
    public RecommendationResponse generateRecommendation(RecommendationRequest request) {

        Long userProfileId = request.userProfileId();

        UserProfile userProfile = userProfileRepository.findById(userProfileId)
                .orElseThrow(() -> new ResourceNotFoundException("User profile not found!"));

        List<Asset> assets = assetRepository.findAll();
        if(assets.isEmpty()) {
            throw new ResourceNotFoundException("No assets founs in the system!");
        }

        Recommendation recommendation = new Recommendation();
        recommendation.setUserProfile(userProfile);
        recommendation.setGeneratedAt(LocalDateTime.now());

        List<Allocation> allocations = new ArrayList<>();
        double percentagePerAsset = 100.0 / assets.size();

        for (Asset asset : assets) {
            Allocation allocation = new Allocation();
            allocation.setAsset(asset);
            allocation.setPercentage(percentagePerAsset);
            allocation.setRecommendation(recommendation);

            allocations.add(allocation);
        }

        recommendation.setAllocations(allocations);

        Recommendation savedRecommendation = recommendationRepository.save(recommendation);

        allocationRepository.saveAll(allocations);

        return RecommendationMapper.toResponse(savedRecommendation);
    }

    public RecommendationResponse getById(Long id) {
        Recommendation recommendation = recommendationRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Recommendation not found!"));
        return RecommendationMapper.toResponse(recommendation);
    }
}
