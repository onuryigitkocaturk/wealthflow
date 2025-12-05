package com.wealthflow.backend.mapper;

import com.wealthflow.backend.dto.AssetResponse;
import com.wealthflow.backend.model.Asset;

public class AssetMapper {

    public static AssetResponse toResponse(Asset asset) {
        return new AssetResponse(
                asset.getId(),
                asset.getSymbol(),
                asset.getName(),
                asset.getAssetType(),
                asset.getRiskLevel(),
                asset.getEsgScore()
        );
    }
}
