package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.dto.AssetResponse;
import com.wealthflow.backend.exception.ResourceNotFoundException;
import com.wealthflow.backend.mapper.AssetMapper;
import com.wealthflow.backend.model.Asset;
import com.wealthflow.backend.repository.AssetRepository;
import com.wealthflow.backend.service.AssetService;
import jakarta.transaction.Transactional;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@Transactional
public class AssetServiceImpl implements AssetService {

    private final AssetRepository assetRepository;

    public AssetServiceImpl(AssetRepository assetRepository) {
        this.assetRepository = assetRepository;
    }

    @Override
    public List<AssetResponse> getAllAssets() {
        return assetRepository.findAll()
                .stream()
                .map(AssetMapper::toResponse)
                .toList();
    }

    @Override
    public AssetResponse getAssetById(Long id) {
        Asset asset = assetRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Asset not found with id: " + id));

        return AssetMapper.toResponse(asset);
    }
}
