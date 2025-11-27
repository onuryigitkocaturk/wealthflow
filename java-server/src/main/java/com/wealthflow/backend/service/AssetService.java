package com.wealthflow.backend.service;

import com.wealthflow.backend.model.Asset;
import java.util.List;

public interface AssetService {

    List<Asset> getAllAssets();
    Asset getAssetById(Long id);
    Asset createAsset(Asset asset);
    void deleteAsset(Long id);
}
