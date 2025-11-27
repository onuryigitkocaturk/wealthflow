package com.wealthflow.backend.service.impl;

import com.wealthflow.backend.exception.ResourceNotFoundException;
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
    public List<Asset> getAllAssets() {
        return assetRepository.findAll();
    }

    @Override
    public Asset getAssetById(Long id) {
        return assetRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Asset not found with id: " + id));
    }

    @Override
    public Asset createAsset(Asset asset) {
        return assetRepository.save(asset);
    }

    @Override
    public void deleteAsset(Long id) {

        if (!assetRepository.existsById((id))) {
            throw new ResourceNotFoundException("Asset not found with id: " + id);
        }

        assetRepository.deleteById(id);
    }
}
