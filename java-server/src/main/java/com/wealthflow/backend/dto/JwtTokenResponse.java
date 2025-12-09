package com.wealthflow.backend.dto;

public record JwtTokenResponse(
        String token,
        long expiresIn
) {
}
