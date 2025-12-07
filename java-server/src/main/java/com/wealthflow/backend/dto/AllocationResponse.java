package com.wealthflow.backend.dto;

public record AllocationResponse(
        String assetSymbol,
        Double percentage
) {}
