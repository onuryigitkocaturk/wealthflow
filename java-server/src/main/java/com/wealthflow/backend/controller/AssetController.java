package com.wealthflow.backend.controller;

import com.wealthflow.backend.dto.AssetResponse;
import com.wealthflow.backend.mapper.AssetMapper;
import com.wealthflow.backend.model.Asset;
import com.wealthflow.backend.service.AssetService;
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
    public ResponseEntity<List<AssetResponse>> getAllAssets() {
        List<AssetResponse> response = assetService.getAllAssets();
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<AssetResponse> getAssetById(@PathVariable Long id) {
        AssetResponse response = assetService.getAssetById(id);
        return ResponseEntity.ok(response);
    }
}