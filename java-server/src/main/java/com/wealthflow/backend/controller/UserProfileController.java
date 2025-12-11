package com.wealthflow.backend.controller;

import com.wealthflow.backend.api.ApiResponse;
import com.wealthflow.backend.api.ApiResponseBuilder;
import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;
import com.wealthflow.backend.service.UserProfileService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserProfileController {

    private final UserProfileService userProfileService;

    public UserProfileController(UserProfileService userProfileService) {
        this.userProfileService = userProfileService;
    }

    @PostMapping
    public ResponseEntity<ApiResponse<UserProfileResponse>> createProfile(
            @RequestBody @Valid UserProfileRequest requestDto,
            HttpServletRequest request
    ) {
        UserProfileResponse response = userProfileService.createProfile(requestDto);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "User profile created successfully", response)
        );
    }

    @GetMapping("/{id}")
    public ResponseEntity<ApiResponse<UserProfileResponse>> getProfile(
            @PathVariable Long id,
            HttpServletRequest request
    ) {
        UserProfileResponse response = userProfileService.getProfile(id);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "User profile fetched successfully", response)
        );
    }

    @PutMapping("/{id}")
    public ResponseEntity<ApiResponse<UserProfileResponse>> updateProfile(
            @PathVariable Long id,
            @RequestBody @Valid UserProfileRequest requestDto,
            HttpServletRequest request
    ) {
        UserProfileResponse response = userProfileService.updateProfile(id, requestDto);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request,"User profile updated successfully", response)
        );
    }

    @GetMapping
    public ResponseEntity<ApiResponse<List<UserProfileResponse>>> getAllProfiles(HttpServletRequest request) {
        List<UserProfileResponse> response = userProfileService.getAllProfiles();
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "All user profiles fetched successfully", response)
        );
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<ApiResponse<Void>> deleteProfile(
            @PathVariable Long id,
            HttpServletRequest request
    ) {
        userProfileService.deleteProfile(id);
        return ResponseEntity.ok(
                ApiResponseBuilder.success(request, "User profile deleted successfully", null)
        );
    }
}
