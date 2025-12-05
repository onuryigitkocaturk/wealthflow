package com.wealthflow.backend.service;

import com.wealthflow.backend.dto.AssetResponse;
import com.wealthflow.backend.model.Asset;
import java.util.List;

public interface AssetService {

    List<AssetResponse> getAllAssets();
    AssetResponse getAssetById(Long id);
//    Asset createAsset(Asset asset);
//    void deleteAsset(Long id);
}
