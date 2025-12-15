package com.wealthflow.backend.controller;

import com.wealthflow.backend.api.ApiResponse;
import com.wealthflow.backend.api.ApiResponseBuilder;
import com.wealthflow.backend.dto.UserProfileRequest;
import com.wealthflow.backend.dto.UserProfileResponse;
import com.wealthflow.backend.service.UserProfileService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserProfileController {

    private final UserProfileService userProfileService;

    public UserProfileController(UserProfileService userProfileService) {
        this.userProfileService = userProfileService;
    }

    @GetMapping("/me")
    public ResponseEntity<ApiResponse<UserProfileResponse>> getMyProfile(
            HttpServletRequest request,
            Authentication authentication
    ) {
        String email = authentication.getName();

        UserProfileResponse response = userProfileService.getProfileByEmail(email);

        return ResponseEntity.ok(
                ApiResponseBuilder.success(
                        request,
                        "User profile fethced successfulyy",
                        response
                )
        );
    }

    @PutMapping("/me")
    public ResponseEntity<ApiResponse<UserProfileResponse>> updateMyProfile(
            @RequestBody @Valid UserProfileRequest requestDto,
            HttpServletRequest request,
            Authentication authentication
    ) {
        String email = authentication.getName();

        UserProfileResponse response = userProfileService.updateProfileByEmail(email, requestDto);

        return ResponseEntity.ok(
                ApiResponseBuilder.success(
                        request,
                        "User profile updated successfully",
                        response
                )
        );
    }
}
