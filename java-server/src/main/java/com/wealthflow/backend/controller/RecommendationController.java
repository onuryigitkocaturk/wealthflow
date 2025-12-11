package com.wealthflow.backend.controller;

import com.wealthflow.backend.api.ApiResponse;
import com.wealthflow.backend.api.ApiResponseBuilder;
import com.wealthflow.backend.dto.RecommendationRequest;
import com.wealthflow.backend.dto.RecommendationResponse;
import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.service.RecommendationService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/recommendations")
public class RecommendationController {

    private final RecommendationService recommendationService;

    public RecommendationController(RecommendationService recommendationService) {
        this.recommendationService = recommendationService;
    }

    @PostMapping("/generate")
    public ResponseEntity<ApiResponse<RecommendationResponse>> generate(
            @RequestBody RecommendationRequest requestDto,
            HttpServletRequest request
    ) {
        RecommendationResponse response = recommendationService.generateRecommendation(requestDto);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "Recommendation generated succesfully", response)
        );
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<RecommendationResponse>> getById(
            @PathVariable Long id,
            HttpServletRequest request
    ) {
        RecommendationResponse response = recommendationService.getById(id);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "Recommendation fetched successfully", response)
        );
    }
}
