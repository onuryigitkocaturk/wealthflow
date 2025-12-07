package com.wealthflow.backend.controller;

import com.wealthflow.backend.dto.RecommendationRequest;
import com.wealthflow.backend.dto.RecommendationResponse;
import com.wealthflow.backend.model.Recommendation;
import com.wealthflow.backend.service.RecommendationService;
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
    public ResponseEntity<RecommendationResponse> generate(@RequestBody RecommendationRequest request) {
        RecommendationResponse response = recommendationService.generateRecommendation(request);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<RecommendationResponse> getById(@PathVariable Long id) {
        RecommendationResponse response = recommendationService.getById(id);
        return ResponseEntity.ok(response);
    }
}
