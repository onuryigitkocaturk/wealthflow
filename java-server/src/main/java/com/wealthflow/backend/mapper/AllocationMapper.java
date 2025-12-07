package com.wealthflow.backend.mapper;

import com.wealthflow.backend.dto.AllocationResponse;
import com.wealthflow.backend.model.Allocation;

public class AllocationMapper {

    public static AllocationResponse toResponse(Allocation entity) {
        return new AllocationResponse(
                entity.getAsset().getSymbol(),
                entity.getPercentage()
        );
    }
}
