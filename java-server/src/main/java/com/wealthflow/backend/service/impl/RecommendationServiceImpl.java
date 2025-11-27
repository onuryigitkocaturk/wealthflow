package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.exception.ResourceNotFoundException;
import com.wealthflow.backend.model.Allocation;
import com.wealthflow.backend.model.Asset;
import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.model.UserProfile;
import com.wealthflow.backend.repository.AllocationRepository;
import com.wealthflow.backend.repository.AssetRepository;
import com.wealthflow.backend.repository.RecommendationRepository;
import com.wealthflow.backend.service.RecommendationService;
import jakarta.annotation.Resource;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;
import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class RecommendationServiceImpl implements RecommendationService {

    private final AssetRepository assetRepository;
    private final RecommendationRepository recommendationRepository;
    private final AllocationRepository allocationRepository;

    public RecommendationServiceImpl(
            AssetRepository assetRepository,
            RecommendationRepository recommendationRepository,
            AllocationRepository allocationRepository
    ) {
        this.assetRepository = assetRepository;
        this.recommendationRepository = recommendationRepository;
        this.allocationRepository = allocationRepository;
    }

    @Override
    public Recommendation generateRecommendation(UserProfile userProfile) {

        if (userProfile == null) {
            throw new ResourceNotFoundException("User profile can not be null");
        }

        List<Asset> assets = assetRepository.findAll();
        if (assets.isEmpty()) {
            throw new ResourceNotFoundException("No assets found in the system");
        }

        Recommendation recommendation = new Recommendation();
        recommendation.setUserProfile(userProfile);

        // Static mock allocation until Python Server is activated
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

        // Save recommendation -> parent
        Recommendation savedRecommendation = recommendationRepository.save(recommendation);

        // Save allocations -> children
        allocationRepository.saveAll(allocations);

        return savedRecommendation;
    }
}
