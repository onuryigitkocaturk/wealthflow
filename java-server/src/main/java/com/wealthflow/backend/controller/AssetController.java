package com.wealthflow.backend.controller;

import com.wealthflow.backend.api.ApiResponse;
import com.wealthflow.backend.api.ApiResponseBuilder;
import com.wealthflow.backend.dto.AssetResponse;
import com.wealthflow.backend.service.AssetService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/assets")
public class AssetController {

    private final AssetService assetService;

    public AssetController(AssetService assetService) {
        this.assetService = assetService;
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<AssetResponse>>> getAllAssets(HttpServletRequest request) {
        List<AssetResponse> assets = assetService.getAllAssets();
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "Asset fetched successfully", assets)
        );
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<AssetResponse>> getAssetById(
            @PathVariable Long id,
            HttpServletRequest request
    ) {
        AssetResponse asset = assetService.getAssetById(id);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "Asset fetched successfully", asset)
        );
    }
}