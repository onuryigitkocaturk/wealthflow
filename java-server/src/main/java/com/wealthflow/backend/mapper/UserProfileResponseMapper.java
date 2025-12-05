package com.wealthflow.backend.mapper;

import com.wealthflow.backend.dto.UserProfileResponse;
import com.wealthflow.backend.model.UserProfile;

public class UserProfileResponseMapper {

    public static UserProfileResponse toResponse(UserProfile p) {
        return new UserProfileResponse(
                p.getId(),
                p.getEmail(),
                p.getName(),
                p.getAge(),
                p.getAnnualIncome(),
                p.getInvestmentHorizon(),
                p.getEsgPreference(),
                p.getRiskTolerance(),
                p.getRiskScore(),
                p.getRiskOverride(),
                p.getOverrideTolerance() == null ? null : p.getOverrideTolerance().toString()
        );
    }

}
