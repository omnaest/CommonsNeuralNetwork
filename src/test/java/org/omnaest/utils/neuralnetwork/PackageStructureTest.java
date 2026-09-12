package org.omnaest.utils.neuralnetwork;

import org.junit.jupiter.api.Test;
import org.omnaest.utils.style.StyleProfile;
import org.omnaest.utils.style.sourcetext.SourceGuard;

/**
 * Mechanically enforces this workspace's Java package structure guideline (see
 * {@code .claude/guidelines/java-package-structure.md}) against {@code CommonsNeuralNetwork} via
 * {@code CommonsStyleSupport}'s LIBRARY profile (plan-188).
 * <p>
 * One {@code @Test} method per check, so a project opts out of one check by deleting one line. This class is
 * test-only structural-assertion infrastructure with no main-source counterpart, so it is exempt from the
 * test-mirror rule (P13) and legitimately sits at the context root.
 */
class PackageStructureTest
{

    private static final StyleProfile PROFILE = StyleProfile.library("org.omnaest.utils.neuralnetwork");

    @Test
    void singleEntryPointAtContextRoot()
    {
        PROFILE.singleEntryPointAtContextRoot()
               .check(PROFILE.mainClasses());
    }

    @Test
    void entryPointIsInterfaceOrUtilsFactory()
    {
        PROFILE.entryPointIsInterfaceOrUtilsFactory()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noHorizontalLayerPackages()
    {
        PROFILE.noHorizontalLayerPackages()
               .check(PROFILE.mainClasses());
    }

    @Test
    void internalPackagesAreAccessedOnlyFromWithinTheirOwnSubtree()
    {
        PROFILE.internalPackagesAreAccessedOnlyFromWithinTheirOwnSubtree()
               .check(PROFILE.mainClasses());
    }

    @Test
    void repositoryTypesLiveInInternalRepository()
    {
        PROFILE.repositoryTypesLiveInInternalRepository()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noDtoTypesOutsideInternal()
    {
        PROFILE.noDtoTypesOutsideInternal()
               .check(PROFILE.mainClasses());
    }

    @Test
    void internalSubPackagesAreRoleNamed()
    {
        PROFILE.internalSubPackagesAreRoleNamed()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noInternalTypeOnAPublicApiSurface()
    {
        PROFILE.noInternalTypeOnAPublicApiSurface()
               .check(PROFILE.mainClasses());
    }

    @Test
    void noInternalReferencesFromOutsideTheirOwnSubtree()
    {
        SourceGuard.of()
                   .noInternalReferencesFromOutsideTheirOwnSubtree()
                   .verify();
    }

    @Test
    void testsMirrorTheirSubjectPackage()
    {
        SourceGuard.of()
                   .testsMirrorTheirSubjectPackage()
                   .verify();
    }

}
