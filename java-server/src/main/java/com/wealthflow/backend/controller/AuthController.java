package com.wealthflow.backend.controller;

import com.wealthflow.backend.api.ApiResponse;
import com.wealthflow.backend.api.ApiResponseBuilder;
import com.wealthflow.backend.dto.LoginRequest;
import com.wealthflow.backend.dto.JwtTokenResponse;
import com.wealthflow.backend.dto.RegisterRequest;
import com.wealthflow.backend.service.AuthService;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/login")
    public ResponseEntity<ApiResponse<JwtTokenResponse>> login(
            @Valid @RequestBody LoginRequest requestDto,
            HttpServletRequest request
    ) {
        JwtTokenResponse token = authService.login(requestDto);
        return ResponseEntity.ok(ApiResponseBuilder.success(request, "Login successful", token)
        );
    }

    @PostMapping("/register")
    public ResponseEntity<ApiResponse<JwtTokenResponse>> register(
            @Valid @RequestBody RegisterRequest requestDto,
            HttpServletRequest request
    ) {
        JwtTokenResponse token = authService.register(requestDto);
        return ResponseEntity.ok(ApiResponseBuilder.success(request, "Registration successful", token)
        );
    }
}
